// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void pack_B_tile_wq_int8_i8mm(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void gemm_transB_packed_tile_wq_int8_i8mm(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
void pack_B_tile_wq_int8_asimddp(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void gemm_transB_packed_tile_wq_int8_asimddp(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        quantize_A_tile_wq_int8_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        quantize_A_tile_wq_int8_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __ARM_NEON
#if __aarch64__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
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
                    float v0 = p0a[0];
                    float v1 = p0a[A_hstep];
                    float v2 = p0a[A_hstep * 2];
                    float v3 = p0a[A_hstep * 3];
                    float v4 = p0a[A_hstep * 4];
                    float v5 = p0a[A_hstep * 5];
                    float v6 = p0a[A_hstep * 6];
                    float v7 = p0a[A_hstep * 7];
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
                    float32x4_t _p0 = vld1q_f32(p0);
                    float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                    float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                    float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);
                    float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 4);
                    float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 5);
                    float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 6);
                    float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 7);
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
                    float v00 = p0[0];
                    float v01 = p0[1];
                    float v10 = p0[A_hstep];
                    float v11 = p0[A_hstep + 1];
                    float v20 = p0[A_hstep * 2];
                    float v21 = p0[A_hstep * 2 + 1];
                    float v30 = p0[A_hstep * 3];
                    float v31 = p0[A_hstep * 3 + 1];
                    float v40 = p0[A_hstep * 4];
                    float v41 = p0[A_hstep * 4 + 1];
                    float v50 = p0[A_hstep * 5];
                    float v51 = p0[A_hstep * 5 + 1];
                    float v60 = p0[A_hstep * 6];
                    float v61 = p0[A_hstep * 6 + 1];
                    float v70 = p0[A_hstep * 7];
                    float v71 = p0[A_hstep * 7 + 1];
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
                    float v0 = p0[0];
                    float v1 = p0[A_hstep];
                    float v2 = p0[A_hstep * 2];
                    float v3 = p0[A_hstep * 3];
                    float v4 = p0[A_hstep * 4];
                    float v5 = p0[A_hstep * 5];
                    float v6 = p0[A_hstep * 6];
                    float v7 = p0[A_hstep * 7];
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
#endif // __aarch64__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
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
                    float v0 = p0a[0];
                    float v1 = p0a[A_hstep];
                    float v2 = p0a[A_hstep * 2];
                    float v3 = p0a[A_hstep * 3];
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
                    float32x4_t _p00 = vld1q_f32(p0);
                    float32x4_t _p01 = vld1q_f32(p0 + 4);
                    float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                    float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
                    float32x4_t _p20 = vld1q_f32(p0 + A_hstep * 2);
                    float32x4_t _p21 = vld1q_f32(p0 + A_hstep * 2 + 4);
                    float32x4_t _p30 = vld1q_f32(p0 + A_hstep * 3);
                    float32x4_t _p31 = vld1q_f32(p0 + A_hstep * 3 + 4);
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
                    float32x4_t _p0 = vld1q_f32(p0);
                    float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                    float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                    float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);
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
                    float v00 = p0[0];
                    float v01 = p0[1];
                    float v10 = p0[A_hstep];
                    float v11 = p0[A_hstep + 1];
                    float v20 = p0[A_hstep * 2];
                    float v21 = p0[A_hstep * 2 + 1];
                    float v30 = p0[A_hstep * 3];
                    float v31 = p0[A_hstep * 3 + 1];
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
                    float v0 = p0[0];
                    float v1 = p0[A_hstep];
                    float v2 = p0[A_hstep * 2];
                    float v3 = p0[A_hstep * 3];
                    *pp++ = float2int8(v0 * scale0);
                    *pp++ = float2int8(v1 * scale1);
                    *pp++ = float2int8(v2 * scale2);
                    *pp++ = float2int8(v3 * scale3);
                    p0++;
                }
                pd += 4;
            }
        }
#endif // __ARM_NEON
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
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
                float v0 = p0a[0];
                float v1 = p0a[A_hstep];
                float v2 = p0a[A_hstep * 2];
                float v3 = p0a[A_hstep * 3];
                float v4 = p0a[A_hstep * 4];
                float v5 = p0a[A_hstep * 5];
                float v6 = p0a[A_hstep * 6];
                float v7 = p0a[A_hstep * 7];
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
                float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _s);
                float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s);
                float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _s);
                float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _s);
                float32x4_t _p4 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _s);
                float32x4_t _p5 = vmulq_f32(vld1q_f32(p0 + A_hstep * 5), _s);
                float32x4_t _p6 = vmulq_f32(vld1q_f32(p0 + A_hstep * 6), _s);
                float32x4_t _p7 = vmulq_f32(vld1q_f32(p0 + A_hstep * 7), _s);
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
                float v00 = p0[0];
                float v01 = p0[1];
                float v10 = p0[A_hstep];
                float v11 = p0[A_hstep + 1];
                float v20 = p0[A_hstep * 2];
                float v21 = p0[A_hstep * 2 + 1];
                float v30 = p0[A_hstep * 3];
                float v31 = p0[A_hstep * 3 + 1];
                float v40 = p0[A_hstep * 4];
                float v41 = p0[A_hstep * 4 + 1];
                float v50 = p0[A_hstep * 5];
                float v51 = p0[A_hstep * 5 + 1];
                float v60 = p0[A_hstep * 6];
                float v61 = p0[A_hstep * 6 + 1];
                float v70 = p0[A_hstep * 7];
                float v71 = p0[A_hstep * 7 + 1];
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
                float v0 = p0[0];
                float v1 = p0[A_hstep];
                float v2 = p0[A_hstep * 2];
                float v3 = p0[A_hstep * 3];
                float v4 = p0[A_hstep * 4];
                float v5 = p0[A_hstep * 5];
                float v6 = p0[A_hstep * 6];
                float v7 = p0[A_hstep * 7];
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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
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
                float v0 = p0a[0];
                float v1 = p0a[A_hstep];
                float v2 = p0a[A_hstep * 2];
                float v3 = p0a[A_hstep * 3];
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
                float32x4_t _p00 = vmulq_f32(vld1q_f32(p0), _s0);
                float32x4_t _p01 = vmulq_f32(vld1q_f32(p0 + 4), _s1);
                float32x4_t _p10 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s0);
                float32x4_t _p11 = vmulq_f32(vld1q_f32(p0 + A_hstep + 4), _s1);
                float32x4_t _p20 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _s0);
                float32x4_t _p21 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2 + 4), _s1);
                float32x4_t _p30 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _s0);
                float32x4_t _p31 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3 + 4), _s1);
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
                float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _s);
                float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s);
                float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _s);
                float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _s);
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
                float v00 = p0[0];
                float v01 = p0[1];
                float v10 = p0[A_hstep];
                float v11 = p0[A_hstep + 1];
                float v20 = p0[A_hstep * 2];
                float v21 = p0[A_hstep * 2 + 1];
                float v30 = p0[A_hstep * 3];
                float v31 = p0[A_hstep * 3 + 1];
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
                float v0 = p0[0];
                float v1 = p0[A_hstep];
                float v2 = p0[A_hstep * 2];
                float v3 = p0[A_hstep * 3];
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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
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

static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_quantize_A_tile_wq_int8_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_quantize_A_tile_wq_int8_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __ARM_NEON
#if __aarch64__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
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
#endif // __aarch64__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                float32x4_t _absmax = vdupq_n_f32(0.f);
                const float* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vld1q_f32(p0a);
                    _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
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
#endif // __ARM_NEON
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
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
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

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

        return;
    }

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = (const float*)input_scales + k;
        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
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
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

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

// Persistent B uses the baseline gemm_int8 K2/K1 byte order inside each
// (output-column panel, quantization group). Every panel is exact and the
// address of the panel starting at output column j is always j * K.
// Two consecutive nr4 panels form the logical nr8 panel on aarch64.
// The non-neon simd32 nr2 panel follows the ordinary gemm_int8 per-k producer.
// Its bytes are b0k0 b1k0 b0k1 b1k1 so sxtb16 and ror+sxtb16 extract the two
// output columns without an extra byte shuffle in the smlad inner loop.
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

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* pC = C;

    const float* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;

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
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
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

            vst1q_f32(p0, _out00);
            vst1q_f32(p0 + 4, _out01);
            vst1q_f32(p0 + out_hstep, _out10);
            vst1q_f32(p0 + out_hstep + 4, _out11);
            vst1q_f32(p0 + out_hstep * 2, _out20);
            vst1q_f32(p0 + out_hstep * 2 + 4, _out21);
            vst1q_f32(p0 + out_hstep * 3, _out30);
            vst1q_f32(p0 + out_hstep * 3 + 4, _out31);

            vst1q_f32(p0 + out_hstep * 4, _out40);
            vst1q_f32(p0 + out_hstep * 4 + 4, _out41);
            vst1q_f32(p0 + out_hstep * 5, _out50);
            vst1q_f32(p0 + out_hstep * 5 + 4, _out51);
            vst1q_f32(p0 + out_hstep * 6, _out60);
            vst1q_f32(p0 + out_hstep * 6 + 4, _out61);
            vst1q_f32(p0 + out_hstep * 7, _out70);
            vst1q_f32(p0 + out_hstep * 7 + 4, _out71);

            pp += 64;
            p0 += 8;
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
            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + out_hstep, _out1);
            vst1q_f32(p0 + out_hstep * 2, _out2);
            vst1q_f32(p0 + out_hstep * 3, _out3);
            vst1q_f32(p0 + out_hstep * 4, _out4);
            vst1q_f32(p0 + out_hstep * 5, _out5);
            vst1q_f32(p0 + out_hstep * 6, _out6);
            vst1q_f32(p0 + out_hstep * 7, _out7);
            pp += 32;
            p0 += 4;
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
            vst1_f32(p0, _out0);
            vst1_f32(p0 + out_hstep, _out1);
            vst1_f32(p0 + out_hstep * 2, _out2);
            vst1_f32(p0 + out_hstep * 3, _out3);
            vst1_f32(p0 + out_hstep * 4, _out4);
            vst1_f32(p0 + out_hstep * 5, _out5);
            vst1_f32(p0 + out_hstep * 6, _out6);
            vst1_f32(p0 + out_hstep * 7, _out7);
            pp += 16;
            p0 += 2;
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
            p0[0] = f0 * alpha;
            p0[out_hstep] = f1 * alpha;
            p0[out_hstep * 2] = f2 * alpha;
            p0[out_hstep * 3] = f3 * alpha;
            p0[out_hstep * 4] = f4 * alpha;
            p0[out_hstep * 5] = f5 * alpha;
            p0[out_hstep * 6] = f6 * alpha;
            p0[out_hstep * 7] = f7 * alpha;
            pp += 8;
            p0++;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;

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
                pC = (const float*)C + (i + ii) * c_hstep + j;
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

            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + 4, _out1);
            vst1q_f32(p0 + out_hstep, _out2);
            vst1q_f32(p0 + out_hstep + 4, _out3);
            vst1q_f32(p0 + out_hstep * 2, _out4);
            vst1q_f32(p0 + out_hstep * 2 + 4, _out5);
            vst1q_f32(p0 + out_hstep * 3, _out6);
            vst1q_f32(p0 + out_hstep * 3 + 4, _out7);

            pp += 32;
            p0 += 8;
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

            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + out_hstep, _out1);
            vst1q_f32(p0 + out_hstep * 2, _out2);
            vst1q_f32(p0 + out_hstep * 3, _out3);

            pp += 16;
            p0 += 4;
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

            vst1_f32(p0, vget_low_f32(_out0));
            vst1_f32(p0 + out_hstep, vget_high_f32(_out0));
            vst1_f32(p0 + out_hstep * 2, vget_low_f32(_out1));
            vst1_f32(p0 + out_hstep * 3, vget_high_f32(_out1));

            pp += 8;
            p0 += 2;
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

            vst1q_lane_f32(p0, _out0, 0);
            vst1q_lane_f32(p0 + out_hstep, _out0, 1);
            vst1q_lane_f32(p0 + out_hstep * 2, _out0, 2);
            vst1q_lane_f32(p0 + out_hstep * 3, _out0, 3);

            pp += 4;
            p0++;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;
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
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
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

            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + 4, _out1);
            vst1q_f32(p0 + out_hstep, _out2);
            vst1q_f32(p0 + out_hstep + 4, _out3);

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

            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + out_hstep, _out1);

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

            vst1_f32(p0, _out0);
            vst1_f32(p0 + out_hstep, _out1);

            pp += 4;
            p0 += 2;
        }
        for (; jj < max_jj; jj += 1)
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
                    float32x2_t _c = vget_low_f32(_c0);
                    _c = vset_lane_f32(vgetq_lane_f32(_c1, 0), _c, 1);
                    _out0 = vadd_f32(_out0, _c);
                }
                if (broadcast_type_C == 3)
                {
                    float32x2_t _c = vdup_n_f32(pC[0]);
                    _c = vset_lane_f32(pC[c_hstep], _c, 1);
                    if (beta == 1.f)
                        _out0 = vadd_f32(_out0, _c);
                    else
                        _out0 = vmla_n_f32(_out0, _c, beta);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(pC[0] * beta));
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
            }

            vst1_lane_f32(p0, _out0, 0);
            vst1_lane_f32(p0 + out_hstep, _out0, 1);

            pp += 2;
            p0++;
        }
#endif // __ARM_NEON
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

        for (; jj + 1 < max_jj; jj += 2)
        {
            float out00 = pp[0];
            float out01 = pp[1];
            float out10 = pp[2];
            float out11 = pp[3];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out01 += c0;
                    out10 += c0;
                    out11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out01 += c0;
                    out10 += c1;
                    out11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += pC[0] * beta;
                    out01 += pC[1] * beta;
                    out10 += pC[c_hstep] * beta;
                    out11 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float cc0 = pC[0] * beta;
                    float cc1 = pC[1] * beta;
                    out00 += cc0;
                    out01 += cc1;
                    out10 += cc0;
                    out11 += cc1;
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out01 *= alpha;
                out10 *= alpha;
                out11 *= alpha;
            }

            p0[0] = out00;
            p0[1] = out01;
            p0[out_hstep] = out10;
            p0[out_hstep + 1] = out11;

            pp += 4;
            p0 += 2;
        }
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

            p0[0] = out00;
            p0[out_hstep] = out10;

            pp += 2;
            p0++;
        }
    }
    for (; ii < max_ii; ii++)
    {
        float* p0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;
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
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
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

            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + 4, _out1);
            vst1q_f32(p0 + 8, _out2);
            vst1q_f32(p0 + 12, _out3);

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

            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + 4, _out1);

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

            vst1q_f32(p0, _out0);

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

            vst1_f32(p0, _out0);

            pp += 2;
            p0 += 2;
        }
#endif // __ARM_NEON
        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[0] * beta;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            float out00 = pp[0];
            float out01 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out01 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out01 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += pC[0] * beta;
                    out01 += pC[1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += pC[0] * beta;
                    out01 += pC[1] * beta;
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out01 *= alpha;
            }

            p0[0] = out00;
            p0[1] = out01;

            pp += 2;
            p0 += 2;
        }
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

            p0[0] = out00;

            pp++;
            p0++;
        }
    }
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* pC = C;

    const float* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;

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
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
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

            transpose4x4_ps(_out0, _out1, _out2, _out3);
            transpose4x4_ps(_out4, _out5, _out6, _out7);
            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + 4, _out4);
            vst1q_f32(p0 + out_hstep, _out1);
            vst1q_f32(p0 + out_hstep + 4, _out5);
            vst1q_f32(p0 + out_hstep * 2, _out2);
            vst1q_f32(p0 + out_hstep * 2 + 4, _out6);
            vst1q_f32(p0 + out_hstep * 3, _out3);
            vst1q_f32(p0 + out_hstep * 3 + 4, _out7);

            transpose4x4_ps(_out8, _out9, _outa, _outb);
            transpose4x4_ps(_outc, _outd, _oute, _outf);
            vst1q_f32(p0 + out_hstep * 4, _out8);
            vst1q_f32(p0 + out_hstep * 4 + 4, _outc);
            vst1q_f32(p0 + out_hstep * 5, _out9);
            vst1q_f32(p0 + out_hstep * 5 + 4, _outd);
            vst1q_f32(p0 + out_hstep * 6, _outa);
            vst1q_f32(p0 + out_hstep * 6 + 4, _oute);
            vst1q_f32(p0 + out_hstep * 7, _outb);
            vst1q_f32(p0 + out_hstep * 7 + 4, _outf);
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
            transpose4x4_ps(_out0, _out1, _out2, _out3);
            transpose4x4_ps(_out4, _out5, _out6, _out7);
            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + 4, _out4);
            vst1q_f32(p0 + out_hstep, _out1);
            vst1q_f32(p0 + out_hstep + 4, _out5);
            vst1q_f32(p0 + out_hstep * 2, _out2);
            vst1q_f32(p0 + out_hstep * 2 + 4, _out6);
            vst1q_f32(p0 + out_hstep * 3, _out3);
            vst1q_f32(p0 + out_hstep * 3 + 4, _out7);
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
            vst1q_f32(p0, _t0.val[0]);
            vst1q_f32(p0 + 4, _t1.val[0]);
            vst1q_f32(p0 + out_hstep, _t0.val[1]);
            vst1q_f32(p0 + out_hstep + 4, _t1.val[1]);
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
            vst1q_f32(p0, _out0);
            vst1q_f32(p0 + 4, _out1);
            pp += 8;
            p0 += out_hstep;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
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
                pC = (const float*)C + (i + ii) * c_hstep + j;
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
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);
            float32x4_t _out4 = vld1q_f32(pp + 16);
            float32x4_t _out5 = vld1q_f32(pp + 20);
            float32x4_t _out6 = vld1q_f32(pp + 24);
            float32x4_t _out7 = vld1q_f32(pp + 28);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                    float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                    float32x4_t _c4 = vld1q_f32(pC + c_hstep * 2);
                    float32x4_t _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                    float32x4_t _c6 = vld1q_f32(pC + c_hstep * 3);
                    float32x4_t _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
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
                    pC += 8;
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

            if (out_hstep != 4)
            {
                transpose4x4_ps(_out0, _out2, _out4, _out6);
                transpose4x4_ps(_out1, _out3, _out5, _out7);
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_hstep != 4)
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

            if (out_hstep == 4)
            {
                float32x4x4_t _r0;
                _r0.val[0] = _out0;
                _r0.val[1] = _out2;
                _r0.val[2] = _out4;
                _r0.val[3] = _out6;
                vst4q_f32(p0, _r0);
                float32x4x4_t _r1;
                _r1.val[0] = _out1;
                _r1.val[1] = _out3;
                _r1.val[2] = _out5;
                _r1.val[3] = _out7;
                vst4q_f32(p0 + out_hstep * 4, _r1);
            }
            else
            {
                vst1q_f32(p0, _out0);
                vst1q_f32(p0 + out_hstep, _out2);
                vst1q_f32(p0 + out_hstep * 2, _out4);
                vst1q_f32(p0 + out_hstep * 3, _out6);
                vst1q_f32(p0 + out_hstep * 4, _out1);
                vst1q_f32(p0 + out_hstep * 5, _out3);
                vst1q_f32(p0 + out_hstep * 6, _out5);
                vst1q_f32(p0 + out_hstep * 7, _out7);
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
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                    float32x4_t _c2 = vld1q_f32(pC + c_hstep * 2);
                    float32x4_t _c3 = vld1q_f32(pC + c_hstep * 3);
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
                    pC += 4;
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

            if (out_hstep != 4)
            {
                transpose4x4_ps(_out0, _out1, _out2, _out3);
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_hstep != 4)
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

            if (out_hstep == 4)
            {
                float32x4x4_t _r;
                _r.val[0] = _out0;
                _r.val[1] = _out1;
                _r.val[2] = _out2;
                _r.val[3] = _out3;
                vst4q_f32(p0, _r);
            }
            else
            {
                vst1q_f32(p0, _out0);
                vst1q_f32(p0 + out_hstep, _out1);
                vst1q_f32(p0 + out_hstep * 2, _out2);
                vst1q_f32(p0 + out_hstep * 3, _out3);
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

            if (out_hstep != 4)
            {
                float32x4x2_t _t = vuzpq_f32(_out0, _out1);
                _out0 = _t.val[0];
                _out1 = _t.val[1];
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_hstep != 4)
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

            if (out_hstep == 4)
            {
                float32x2x4_t _r;
                _r.val[0] = vget_low_f32(_out0);
                _r.val[1] = vget_high_f32(_out0);
                _r.val[2] = vget_low_f32(_out1);
                _r.val[3] = vget_high_f32(_out1);
                vst4_f32(p0, _r);
            }
            else
            {
                vst1q_f32(p0, _out0);
                vst1q_f32(p0 + out_hstep, _out1);
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

            vst1q_f32(p0, _out0);

            pp += 4;
            p0 += out_hstep;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
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
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
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

            if (out_hstep != 2)
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
                    if (broadcast_type_C == 0 || out_hstep != 2)
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

            if (out_hstep == 2)
            {
                float32x4x2_t _r0;
                _r0.val[0] = _out0;
                _r0.val[1] = _out2;
                vst2q_f32(p0, _r0);
                float32x4x2_t _r1;
                _r1.val[0] = _out1;
                _r1.val[1] = _out3;
                vst2q_f32(p0 + out_hstep * 4, _r1);
            }
            else
            {
                vst1_f32(p0, vget_low_f32(_out0));
                vst1_f32(p0 + out_hstep, vget_high_f32(_out0));
                vst1_f32(p0 + out_hstep * 2, vget_low_f32(_out1));
                vst1_f32(p0 + out_hstep * 3, vget_high_f32(_out1));
                vst1_f32(p0 + out_hstep * 4, vget_low_f32(_out2));
                vst1_f32(p0 + out_hstep * 5, vget_high_f32(_out2));
                vst1_f32(p0 + out_hstep * 6, vget_low_f32(_out3));
                vst1_f32(p0 + out_hstep * 7, vget_high_f32(_out3));
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

            if (out_hstep != 2)
            {
                float32x4x2_t _t = vzipq_f32(_out0, _out1);
                _out0 = _t.val[0];
                _out1 = _t.val[1];
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_hstep != 2)
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

            if (out_hstep == 2)
            {
                float32x4x2_t _r;
                _r.val[0] = _out0;
                _r.val[1] = _out1;
                vst2q_f32(p0, _r);
            }
            else
            {
                vst1_f32(p0, vget_low_f32(_out0));
                vst1_f32(p0 + out_hstep, vget_high_f32(_out0));
                vst1_f32(p0 + out_hstep * 2, vget_low_f32(_out1));
                vst1_f32(p0 + out_hstep * 3, vget_high_f32(_out1));
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

            if (out_hstep != 2)
            {
                float32x2x2_t _t = vzip_f32(_out0, _out1);
                _out0 = _t.val[0];
                _out1 = _t.val[1];
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_hstep != 2)
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

            if (out_hstep == 2)
            {
                float32x2x2_t _r;
                _r.val[0] = _out0;
                _r.val[1] = _out1;
                vst2_f32(p0, _r);
            }
            else
            {
                vst1_f32(p0, _out0);
                vst1_f32(p0 + out_hstep, _out1);
            }

            pp += 4;
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x2_t _out0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c01));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c01));
                }
                if (broadcast_type_C == 3)
                {
                    float32x2_t _c = vdup_n_f32(pC[0]);
                    _c = vset_lane_f32(pC[c_hstep], _c, 1);
                    if (beta == 1.f)
                        _out0 = vadd_f32(_out0, _c);
                    else
                        _out0 = vmla_n_f32(_out0, _c, beta);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(pC[0] * beta));
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
            }

            vst1_f32(p0, _out0);

            pp += 2;
            p0 += out_hstep;
        }
#endif // __ARM_NEON
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

        for (; jj + 1 < max_jj; jj += 2)
        {
            float out00 = pp[0];
            float out01 = pp[1];
            float out10 = pp[2];
            float out11 = pp[3];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out01 += c0;
                    out10 += c0;
                    out11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out01 += c0;
                    out10 += c1;
                    out11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += pC[0] * beta;
                    out01 += pC[1] * beta;
                    out10 += pC[c_hstep] * beta;
                    out11 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float cc0 = pC[0] * beta;
                    float cc1 = pC[1] * beta;
                    out00 += cc0;
                    out01 += cc1;
                    out10 += cc0;
                    out11 += cc1;
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out01 *= alpha;
                out10 *= alpha;
                out11 *= alpha;
            }

            p0[0] = out00;
            p0[out_hstep] = out01;
            p0[1] = out10;
            p0[out_hstep + 1] = out11;

            pp += 4;
            p0 += out_hstep * 2;
        }
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

            p0[0] = out00;
            p0[1] = out10;

            pp += 2;
            p0 += out_hstep;
        }
    }
    for (; ii < max_ii; ii++)
    {
        float* p0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
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
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
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

            if (out_hstep == 1)
            {
                vst1q_f32(p0, _out0);
                vst1q_f32(p0 + 4, _out1);
                vst1q_f32(p0 + 8, _out2);
                vst1q_f32(p0 + 12, _out3);
            }
            else
            {
                vst1q_lane_f32(p0, _out0, 0);
                vst1q_lane_f32(p0 + out_hstep, _out0, 1);
                vst1q_lane_f32(p0 + out_hstep * 2, _out0, 2);
                vst1q_lane_f32(p0 + out_hstep * 3, _out0, 3);
                vst1q_lane_f32(p0 + out_hstep * 4, _out1, 0);
                vst1q_lane_f32(p0 + out_hstep * 5, _out1, 1);
                vst1q_lane_f32(p0 + out_hstep * 6, _out1, 2);
                vst1q_lane_f32(p0 + out_hstep * 7, _out1, 3);
                vst1q_lane_f32(p0 + out_hstep * 8, _out2, 0);
                vst1q_lane_f32(p0 + out_hstep * 9, _out2, 1);
                vst1q_lane_f32(p0 + out_hstep * 10, _out2, 2);
                vst1q_lane_f32(p0 + out_hstep * 11, _out2, 3);
                vst1q_lane_f32(p0 + out_hstep * 12, _out3, 0);
                vst1q_lane_f32(p0 + out_hstep * 13, _out3, 1);
                vst1q_lane_f32(p0 + out_hstep * 14, _out3, 2);
                vst1q_lane_f32(p0 + out_hstep * 15, _out3, 3);
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

            if (out_hstep == 1)
            {
                vst1q_f32(p0, _out0);
                vst1q_f32(p0 + 4, _out1);
            }
            else
            {
                vst1q_lane_f32(p0, _out0, 0);
                vst1q_lane_f32(p0 + out_hstep, _out0, 1);
                vst1q_lane_f32(p0 + out_hstep * 2, _out0, 2);
                vst1q_lane_f32(p0 + out_hstep * 3, _out0, 3);
                vst1q_lane_f32(p0 + out_hstep * 4, _out1, 0);
                vst1q_lane_f32(p0 + out_hstep * 5, _out1, 1);
                vst1q_lane_f32(p0 + out_hstep * 6, _out1, 2);
                vst1q_lane_f32(p0 + out_hstep * 7, _out1, 3);
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

            if (out_hstep == 1)
            {
                vst1q_f32(p0, _out0);
            }
            else
            {
                vst1q_lane_f32(p0, _out0, 0);
                vst1q_lane_f32(p0 + out_hstep, _out0, 1);
                vst1q_lane_f32(p0 + out_hstep * 2, _out0, 2);
                vst1q_lane_f32(p0 + out_hstep * 3, _out0, 3);
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

            if (out_hstep == 1)
            {
                vst1_f32(p0, _out0);
            }
            else
            {
                vst1_lane_f32(p0, _out0, 0);
                vst1_lane_f32(p0 + out_hstep, _out0, 1);
            }

            pp += 2;
            p0 += out_hstep * 2;
        }
#endif // __ARM_NEON
        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[0] * beta;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            float out00 = pp[0];
            float out01 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out01 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out01 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += pC[0] * beta;
                    out01 += pC[1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += pC[0] * beta;
                    out01 += pC[1] * beta;
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out01 *= alpha;
            }

            p0[0] = out00;
            p0[out_hstep] = out01;

            pp += 2;
            p0 += out_hstep * 2;
        }
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

            p0[0] = out00;

            pp++;
            p0 += out_hstep;
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
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                int32x4_t _msum4 = vdupq_n_s32(0);
                int32x4_t _msum5 = vdupq_n_s32(0);
                int32x4_t _msum6 = vdupq_n_s32(0);
                int32x4_t _msum7 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x16_t _a45 = vld1q_s8(pA + 32);
                    int8x16_t _a67 = vld1q_s8(pA + 48);
                    _msum0 = vmmlaq_s32(_msum0, _a01, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a23, _b0);
                    _msum2 = vmmlaq_s32(_msum2, _a01, _b1);
                    _msum3 = vmmlaq_s32(_msum3, _a23, _b1);
                    _msum4 = vmmlaq_s32(_msum4, _a45, _b0);
                    _msum5 = vmmlaq_s32(_msum5, _a67, _b0);
                    _msum6 = vmmlaq_s32(_msum6, _a45, _b1);
                    _msum7 = vmmlaq_s32(_msum7, _a67, _b1);
                    pA += 64;
                    pB += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum2));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum2));
                _sum2 = vcombine_s32(vget_low_s32(_msum1), vget_low_s32(_msum3));
                _sum3 = vcombine_s32(vget_high_s32(_msum1), vget_high_s32(_msum3));
                _sum4 = vcombine_s32(vget_low_s32(_msum4), vget_low_s32(_msum6));
                _sum5 = vcombine_s32(vget_high_s32(_msum4), vget_high_s32(_msum6));
                _sum6 = vcombine_s32(vget_low_s32(_msum5), vget_low_s32(_msum7));
                _sum7 = vcombine_s32(vget_high_s32(_msum5), vget_high_s32(_msum7));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a0, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a0, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a0, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _b0, _a1, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _b0, _a1, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _b0, _a1, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _b0, _a1, 3);
                    pA += 32;
                    pB += 16;
                }
#else // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x8_t _a01 = vreinterpretq_s16_s8(vld1q_s8(pA));
                    int16x8_t _a23 = vreinterpretq_s16_s8(vld1q_s8(pA + 16));
                    int8x16_t _b = vld1q_s8(pB);
                    int8x8_t _b01 = vget_low_s8(_b);
                    int8x8_t _b23 = vget_high_s8(_b);
                    int16x4_t _a010 = vget_low_s16(_a01);
                    int16x4_t _a011 = vget_high_s16(_a01);
                    int16x4_t _a230 = vget_low_s16(_a23);
                    int16x4_t _a231 = vget_high_s16(_a23);
                    int16x8_t _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a010, 0)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a230, 0)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a010, 1)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a230, 1)));
                    _sum1 = vpadalq_s16(_sum1, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a010, 2)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a230, 2)));
                    _sum2 = vpadalq_s16(_sum2, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a010, 3)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a230, 3)));
                    _sum3 = vpadalq_s16(_sum3, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a011, 0)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a231, 0)));
                    _sum4 = vpadalq_s16(_sum4, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a011, 1)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a231, 1)));
                    _sum5 = vpadalq_s16(_sum5, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a011, 2)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a231, 2)));
                    _sum6 = vpadalq_s16(_sum6, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a011, 3)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a231, 3)));
                    _sum7 = vpadalq_s16(_sum7, _s);
                    pA += 32;
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _b0 = vld1_s8(pB);
                    int16x8_t _a = vreinterpretq_s16_s8(vld1q_s8(pA));
                    int16x4_t _a0 = vget_low_s16(_a);
                    int16x4_t _a1 = vget_high_s16(_a);
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 1)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 2)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 3)))));
                    _sum4 = vaddq_s32(_sum4, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 0)))));
                    _sum5 = vaddq_s32(_sum5, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 1)))));
                    _sum6 = vaddq_s32(_sum6, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 2)))));
                    _sum7 = vaddq_s32(_sum7, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 3)))));
                    pA += 16;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                    int8x8_t _a = vld1_s8(pA);
                    int16x8_t _p0 = vmull_s8(_b0, vdup_lane_s8(_a, 0));
                    int16x8_t _p1 = vmull_s8(_b0, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    int16x8_t _p2 = vmull_s8(_b0, vdup_lane_s8(_a, 2));
                    int16x8_t _p3 = vmull_s8(_b0, vdup_lane_s8(_a, 3));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p2)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p3)));
                    int16x8_t _p4 = vmull_s8(_b0, vdup_lane_s8(_a, 4));
                    int16x8_t _p5 = vmull_s8(_b0, vdup_lane_s8(_a, 5));
                    _sum4 = vaddq_s32(_sum4, vmovl_s16(vget_low_s16(_p4)));
                    _sum5 = vaddq_s32(_sum5, vmovl_s16(vget_low_s16(_p5)));
                    int16x8_t _p6 = vmull_s8(_b0, vdup_lane_s8(_a, 6));
                    int16x8_t _p7 = vmull_s8(_b0, vdup_lane_s8(_a, 7));
                    _sum6 = vaddq_s32(_sum6, vmovl_s16(vget_low_s16(_p6)));
                    _sum7 = vaddq_s32(_sum7, vmovl_s16(vget_low_s16(_p7)));
                    pA += 8;
                    pB += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales);
                float32x4_t _ad0 = vld1q_f32(pA_descales);
                float32x4_t _ad1 = vld1q_f32(pA_descales + 4);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_laneq_f32(_bd0, _ad0, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_laneq_f32(_bd0, _ad0, 1));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_laneq_f32(_bd0, _ad0, 2));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_laneq_f32(_bd0, _ad0, 3));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_laneq_f32(_bd0, _ad1, 0));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_laneq_f32(_bd0, _ad1, 1));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_laneq_f32(_bd0, _ad1, 2));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_laneq_f32(_bd0, _ad1, 3));
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
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x16_t _a45 = vld1q_s8(pA + 32);
                    int8x16_t _a67 = vld1q_s8(pA + 48);
                    int8x16_t _b = vld1q_s8(pB);
                    _s0 = vmmlaq_s32(_s0, _a01, _b);
                    _s1 = vmmlaq_s32(_s1, _a23, _b);
                    _s2 = vmmlaq_s32(_s2, _a45, _b);
                    _s3 = vmmlaq_s32(_s3, _a67, _b);
                    pA += 64;
                    pB += 16;
                }
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
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int16x4_t _b = vreinterpret_s16_s8(vld1_s8(pB));
                    int8x8_t _b0 = vreinterpret_s8_s16(vdup_lane_s16(_b, 0));
                    int8x8_t _b1 = vreinterpret_s8_s16(vdup_lane_s16(_b, 1));
                    int8x8_t _b2 = vreinterpret_s8_s16(vdup_lane_s16(_b, 2));
                    int8x8_t _b3 = vreinterpret_s8_s16(vdup_lane_s16(_b, 3));
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(vget_low_s8(_a0), _b0));
                    _sum1 = vpadalq_s16(_sum1, vmull_s8(vget_low_s8(_a0), _b1));
                    _sum2 = vpadalq_s16(_sum2, vmull_s8(vget_high_s8(_a0), _b0));
                    _sum3 = vpadalq_s16(_sum3, vmull_s8(vget_high_s8(_a0), _b1));
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(vget_low_s8(_a1), _b2));
                    _sum1 = vpadalq_s16(_sum1, vmull_s8(vget_low_s8(_a1), _b3));
                    _sum2 = vpadalq_s16(_sum2, vmull_s8(vget_high_s8(_a1), _b2));
                    _sum3 = vpadalq_s16(_sum3, vmull_s8(vget_high_s8(_a1), _b3));
                    pA += 32;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
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
                    pA += 16;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    int8x8x2_t _b01 = vuzp_s8(_b, _b);
                    int16x8_t _s0 = vmull_s8(_a, _b01.val[0]);
                    int16x8_t _s1 = vmull_s8(_a, _b01.val[1]);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_low_s16(_s1));
                    _sum2 = vaddw_s16(_sum2, vget_high_s16(_s0));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
                    pA += 8;
                    pB += 2;
                }

                float32x2_t _bd = vld1_f32(pB_descales);
                float32x4_t _ad0 = vld1q_f32(pA_descales);
                float32x4_t _ad1 = vld1q_f32(pA_descales + 4);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_ad0, _bd, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_ad0, _bd, 1));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_lane_f32(_ad1, _bd, 0));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_lane_f32(_ad1, _bd, 1));
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
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x16_t _a45 = vld1q_s8(pA + 32);
                    int8x16_t _a67 = vld1q_s8(pA + 48);
                    int8x8_t _b = vld1_s8(pB);
                    int8x16_t _bb = vcombine_s8(_b, _b);
                    int32x4_t _s0 = vdotq_s32(vdupq_n_s32(0), _a01, _bb);
                    int32x4_t _s1 = vdotq_s32(vdupq_n_s32(0), _a23, _bb);
                    int32x4_t _s2 = vdotq_s32(vdupq_n_s32(0), _a45, _bb);
                    int32x4_t _s3 = vdotq_s32(vdupq_n_s32(0), _a67, _bb);
                    _sum0 = vaddq_s32(_sum0, vpaddq_s32(_s0, _s1));
                    _sum1 = vaddq_s32(_sum1, vpaddq_s32(_s2, _s3));
                    pA += 64;
                    pB += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
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
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB_panel + (size_t)4 * k;
            const signed char* pB1 = pB_panel + (size_t)4 * K + (size_t)4 * k;
            const float* pB_descales0 = pB_descales_panel + (size_t)4 * block_start;
            const float* pB_descales1 = pB_descales_panel + (size_t)4 * block_count + (size_t)4 * block_start;
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

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
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
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                int32x4_t _msum4 = vdupq_n_s32(0);
                int32x4_t _msum5 = vdupq_n_s32(0);
                int32x4_t _msum6 = vdupq_n_s32(0);
                int32x4_t _msum7 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x16_t _b00 = vld1q_s8(pB0);
                    int8x16_t _b01 = vld1q_s8(pB0 + 16);
                    int8x16_t _b10 = vld1q_s8(pB1);
                    int8x16_t _b11 = vld1q_s8(pB1 + 16);
                    _msum0 = vmmlaq_s32(_msum0, _a01, _b00);
                    _msum1 = vmmlaq_s32(_msum1, _a01, _b01);
                    _msum2 = vmmlaq_s32(_msum2, _a23, _b00);
                    _msum3 = vmmlaq_s32(_msum3, _a23, _b01);
                    _msum4 = vmmlaq_s32(_msum4, _a01, _b10);
                    _msum5 = vmmlaq_s32(_msum5, _a01, _b11);
                    _msum6 = vmmlaq_s32(_msum6, _a23, _b10);
                    _msum7 = vmmlaq_s32(_msum7, _a23, _b11);
                    pA += 32;
                    pB0 += 32;
                    pB1 += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_low_s32(_msum4), vget_low_s32(_msum5));
                _sum2 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum1));
                _sum3 = vcombine_s32(vget_high_s32(_msum4), vget_high_s32(_msum5));
                _sum4 = vcombine_s32(vget_low_s32(_msum2), vget_low_s32(_msum3));
                _sum5 = vcombine_s32(vget_low_s32(_msum6), vget_low_s32(_msum7));
                _sum6 = vcombine_s32(vget_high_s32(_msum2), vget_high_s32(_msum3));
                _sum7 = vcombine_s32(vget_high_s32(_msum6), vget_high_s32(_msum7));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x16_t _b0 = vld1q_s8(pB0);
                    int8x16_t _b1 = vld1q_s8(pB1);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b1, _a, 0);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a, 1);
                    _sum3 = vdotq_laneq_s32(_sum3, _b1, _a, 1);
                    _sum4 = vdotq_laneq_s32(_sum4, _b0, _a, 2);
                    _sum5 = vdotq_laneq_s32(_sum5, _b1, _a, 2);
                    _sum6 = vdotq_laneq_s32(_sum6, _b0, _a, 3);
                    _sum7 = vdotq_laneq_s32(_sum7, _b1, _a, 3);
                    pA += 16;
                    pB0 += 16;
                    pB1 += 16;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x8_t _a = vreinterpretq_s16_s8(vld1q_s8(pA));
                    int16x4_t _a01 = vget_low_s16(_a);
                    int16x4_t _a23 = vget_high_s16(_a);
                    int8x16_t _b0 = vld1q_s8(pB0);
                    int8x16_t _b1 = vld1q_s8(pB1);
                    int8x8_t _b001 = vget_low_s8(_b0);
                    int8x8_t _b023 = vget_high_s8(_b0);
                    int8x8_t _b101 = vget_low_s8(_b1);
                    int8x8_t _b123 = vget_high_s8(_b1);
                    int16x8_t _s = vmull_s8(_b001, vreinterpret_s8_s16(vdup_lane_s16(_a01, 0)));
                    _s = vmlal_s8(_s, _b023, vreinterpret_s8_s16(vdup_lane_s16(_a23, 0)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    _s = vmull_s8(_b101, vreinterpret_s8_s16(vdup_lane_s16(_a01, 0)));
                    _s = vmlal_s8(_s, _b123, vreinterpret_s8_s16(vdup_lane_s16(_a23, 0)));
                    _sum1 = vpadalq_s16(_sum1, _s);
                    _s = vmull_s8(_b001, vreinterpret_s8_s16(vdup_lane_s16(_a01, 1)));
                    _s = vmlal_s8(_s, _b023, vreinterpret_s8_s16(vdup_lane_s16(_a23, 1)));
                    _sum2 = vpadalq_s16(_sum2, _s);
                    _s = vmull_s8(_b101, vreinterpret_s8_s16(vdup_lane_s16(_a01, 1)));
                    _s = vmlal_s8(_s, _b123, vreinterpret_s8_s16(vdup_lane_s16(_a23, 1)));
                    _sum3 = vpadalq_s16(_sum3, _s);
                    _s = vmull_s8(_b001, vreinterpret_s8_s16(vdup_lane_s16(_a01, 2)));
                    _s = vmlal_s8(_s, _b023, vreinterpret_s8_s16(vdup_lane_s16(_a23, 2)));
                    _sum4 = vpadalq_s16(_sum4, _s);
                    _s = vmull_s8(_b101, vreinterpret_s8_s16(vdup_lane_s16(_a01, 2)));
                    _s = vmlal_s8(_s, _b123, vreinterpret_s8_s16(vdup_lane_s16(_a23, 2)));
                    _sum5 = vpadalq_s16(_sum5, _s);
                    _s = vmull_s8(_b001, vreinterpret_s8_s16(vdup_lane_s16(_a01, 3)));
                    _s = vmlal_s8(_s, _b023, vreinterpret_s8_s16(vdup_lane_s16(_a23, 3)));
                    _sum6 = vpadalq_s16(_sum6, _s);
                    _s = vmull_s8(_b101, vreinterpret_s8_s16(vdup_lane_s16(_a01, 3)));
                    _s = vmlal_s8(_s, _b123, vreinterpret_s8_s16(vdup_lane_s16(_a23, 3)));
                    _sum7 = vpadalq_s16(_sum7, _s);
                    pA += 16;
                    pB0 += 16;
                    pB1 += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    int8x8_t _b0 = vld1_s8(pB0);
                    int8x8_t _b1 = vld1_s8(pB1);
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum4 = vaddq_s32(_sum4, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)))));
                    _sum5 = vaddq_s32(_sum5, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)))));
                    _sum6 = vaddq_s32(_sum6, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)))));
                    _sum7 = vaddq_s32(_sum7, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)))));
                    pA += 8;
                    pB0 += 8;
                    pB1 += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vreinterpret_s8_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB0));
                    int8x8_t _b1 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB1));
                    int16x8_t _p00 = vmull_s8(_b0, vdup_lane_s8(_a, 0));
                    int16x8_t _p01 = vmull_s8(_b1, vdup_lane_s8(_a, 0));
                    int16x8_t _p10 = vmull_s8(_b0, vdup_lane_s8(_a, 1));
                    int16x8_t _p11 = vmull_s8(_b1, vdup_lane_s8(_a, 1));
                    int16x8_t _p20 = vmull_s8(_b0, vdup_lane_s8(_a, 2));
                    int16x8_t _p21 = vmull_s8(_b1, vdup_lane_s8(_a, 2));
                    int16x8_t _p30 = vmull_s8(_b0, vdup_lane_s8(_a, 3));
                    int16x8_t _p31 = vmull_s8(_b1, vdup_lane_s8(_a, 3));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p00)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p01)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p10)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p11)));
                    _sum4 = vaddq_s32(_sum4, vmovl_s16(vget_low_s16(_p20)));
                    _sum5 = vaddq_s32(_sum5, vmovl_s16(vget_low_s16(_p21)));
                    _sum6 = vaddq_s32(_sum6, vmovl_s16(vget_low_s16(_p30)));
                    _sum7 = vaddq_s32(_sum7, vmovl_s16(vget_low_s16(_p31)));
                    pA += 4;
                    pB0 += 4;
                    pB1 += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales0);
                float32x4_t _bd1 = vld1q_f32(pB_descales1);
                float32x4_t _ad = vld1q_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_laneq_f32(_bd0, _ad, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_laneq_f32(_bd1, _ad, 0));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_laneq_f32(_bd0, _ad, 1));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_laneq_f32(_bd1, _ad, 1));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_laneq_f32(_bd0, _ad, 2));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_laneq_f32(_bd1, _ad, 2));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_laneq_f32(_bd0, _ad, 3));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_laneq_f32(_bd1, _ad, 3));

                pA_descales += 4;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            vst1q_f32(outptr, _fsum2);
            outptr += 4;
            vst1q_f32(outptr, _fsum3);
            outptr += 4;
            vst1q_f32(outptr, _fsum4);
            outptr += 4;
            vst1q_f32(outptr, _fsum5);
            outptr += 4;
            vst1q_f32(outptr, _fsum6);
            outptr += 4;
            vst1q_f32(outptr, _fsum7);
            outptr += 4;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __aarch64__
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
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    _msum0 = vmmlaq_s32(_msum0, _a01, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a23, _b0);
                    _msum2 = vmmlaq_s32(_msum2, _a01, _b1);
                    _msum3 = vmmlaq_s32(_msum3, _a23, _b1);
                    pA += 32;
                    pB += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum2));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum2));
                _sum2 = vcombine_s32(vget_low_s32(_msum1), vget_low_s32(_msum3));
                _sum3 = vcombine_s32(vget_high_s32(_msum1), vget_high_s32(_msum3));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _a = vld1q_s8(pA);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a, 3);
                    pA += 16;
                    pB += 16;
                }
#else // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x8_t _a = vreinterpretq_s16_s8(vld1q_s8(pA));
                    int16x4_t _a01 = vget_low_s16(_a);
                    int16x4_t _a23 = vget_high_s16(_a);
                    int8x16_t _b = vld1q_s8(pB);
                    int8x8_t _b01 = vget_low_s8(_b);
                    int8x8_t _b23 = vget_high_s8(_b);
                    int16x8_t _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a01, 0)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a23, 0)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a01, 1)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a23, 1)));
                    _sum1 = vpadalq_s16(_sum1, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a01, 2)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a23, 2)));
                    _sum2 = vpadalq_s16(_sum2, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a01, 3)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a23, 3)));
                    _sum3 = vpadalq_s16(_sum3, _s);
                    pA += 16;
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _b0 = vld1_s8(pB);
                    int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)))));
                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    int8x8_t _a = vreinterpret_s8_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    int16x8_t _p2 = vmull_s8(_b, vdup_lane_s8(_a, 2));
                    int16x8_t _p3 = vmull_s8(_b, vdup_lane_s8(_a, 3));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p2)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p3)));
                    pA += 4;
                    pB += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales);
                float32x4_t _ad = vld1q_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 0)));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 1)));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 2)));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 3)));

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
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x16_t _b = vld1q_s8(pB);
                    _sum0 = vmmlaq_s32(_sum0, _a01, _b);
                    _sum1 = vmmlaq_s32(_sum1, _a23, _b);
                    pA += 32;
                    pB += 16;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = vld1_s8(pB);
                    int32x4_t _s0 = vdotq_lane_s32(vdupq_n_s32(0), _a, _b, 0);
                    int32x4_t _s1 = vdotq_lane_s32(vdupq_n_s32(0), _a, _b, 1);
                    int32x4x2_t _s01 = vzipq_s32(_s0, _s1);
                    _sum0 = vaddq_s32(_sum0, _s01.val[0]);
                    _sum1 = vaddq_s32(_sum1, _s01.val[1]);
                    pA += 16;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a0 = vld1_s8(pA);
                    int8x8_t _a1 = vld1_s8(pA + 8);
                    int16x4_t _b = vreinterpret_s16_s8(vld1_s8(pB));
                    int8x8_t _b0 = vreinterpret_s8_s16(vdup_lane_s16(_b, 0));
                    int8x8_t _b1 = vreinterpret_s8_s16(vdup_lane_s16(_b, 1));
                    int8x8_t _b2 = vreinterpret_s8_s16(vdup_lane_s16(_b, 2));
                    int8x8_t _b3 = vreinterpret_s8_s16(vdup_lane_s16(_b, 3));
                    int32x4_t _s0 = vpaddlq_s16(vmull_s8(_a0, _b0));
                    int32x4_t _s1 = vpaddlq_s16(vmull_s8(_a0, _b1));
                    _s0 = vpadalq_s16(_s0, vmull_s8(_a1, _b2));
                    _s1 = vpadalq_s16(_s1, vmull_s8(_a1, _b3));
                    int32x4x2_t _s01 = vzipq_s32(_s0, _s1);
                    _sum0 = vaddq_s32(_sum0, _s01.val[0]);
                    _sum1 = vaddq_s32(_sum1, _s01.val[1]);
                    pA += 16;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int16x4_t _b = vreinterpret_s16_s32(vld1_dup_s32((const int*)pB));
                    int16x4x2_t _b01 = vuzp_s16(_b, _b);
                    int32x4_t _s0 = vpaddlq_s16(vmull_s8(_a, vreinterpret_s8_s16(_b01.val[0])));
                    int32x4_t _s1 = vpaddlq_s16(vmull_s8(_a, vreinterpret_s8_s16(_b01.val[1])));
                    int32x4x2_t _s01 = vzipq_s32(_s0, _s1);
                    _sum0 = vaddq_s32(_sum0, _s01.val[0]);
                    _sum1 = vaddq_s32(_sum1, _s01.val[1]);
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int32x4_t _s0 = vmovl_s16(vget_low_s16(vmull_s8(_a, vdup_n_s8(pB[0]))));
                    int32x4_t _s1 = vmovl_s16(vget_low_s16(vmull_s8(_a, vdup_n_s8(pB[1]))));
                    int32x4x2_t _s01 = vzipq_s32(_s0, _s1);
                    _sum0 = vaddq_s32(_sum0, _s01.val[0]);
                    _sum1 = vaddq_s32(_sum1, _s01.val[1]);
                    pA += 4;
                    pB += 2;
                }

                float32x2_t _bd = vld1_f32(pB_descales);
                float32x4_t _bdbd = vcombine_f32(_bd, _bd);
                float32x4_t _ad = vld1q_f32(pA_descales);
                float32x4x2_t _ad01 = vzipq_f32(_ad, _ad);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_bdbd, _ad01.val[0]));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_bdbd, _ad01.val[1]));
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
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x8_t _b = vld1_s8(pB);
                    int8x16_t _bb = vcombine_s8(_b, _b);
                    int32x4_t _s0 = vdotq_s32(vdupq_n_s32(0), _a01, _bb);
                    int32x4_t _s1 = vdotq_s32(vdupq_n_s32(0), _a23, _bb);
                    _sum = vaddq_s32(_sum, vcombine_s32(vpadd_s32(vget_low_s32(_s0), vget_high_s32(_s0)), vpadd_s32(vget_low_s32(_s1), vget_high_s32(_s1))));
                    pA += 32;
                    pB += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
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
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB_panel + (size_t)4 * k;
            const signed char* pB1 = pB_panel + (size_t)4 * K + (size_t)4 * k;
            const float* pB_descales0 = pB_descales_panel + (size_t)4 * block_start;
            const float* pB_descales1 = pB_descales_panel + (size_t)4 * block_count + (size_t)4 * block_start;
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
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = vld1q_s8(pB0);
                    int8x16_t _b1 = vld1q_s8(pB0 + 16);
                    int8x16_t _b2 = vld1q_s8(pB1);
                    int8x16_t _b3 = vld1q_s8(pB1 + 16);
                    int8x16_t _a0 = vld1q_s8(pA);
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
                    _msum2 = vmmlaq_s32(_msum2, _a0, _b2);
                    _msum3 = vmmlaq_s32(_msum3, _a0, _b3);
                    pA += 16;
                    pB0 += 32;
                    pB1 += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_low_s32(_msum2), vget_low_s32(_msum3));
                _sum2 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum1));
                _sum3 = vcombine_s32(vget_high_s32(_msum2), vget_high_s32(_msum3));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = vld1q_s8(pB0);
                    int8x16_t _b1 = vld1q_s8(pB1);
                    int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b1, _a, 0);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a, 1);
                    _sum3 = vdotq_laneq_s32(_sum3, _b1, _a, 1);
                    pA += 8;
                    pB0 += 16;
                    pB1 += 16;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    int8x16_t _b0 = vld1q_s8(pB0);
                    int8x16_t _b1 = vld1q_s8(pB1);
                    int8x8_t _b001 = vget_low_s8(_b0);
                    int8x8_t _b023 = vget_high_s8(_b0);
                    int8x8_t _b101 = vget_low_s8(_b1);
                    int8x8_t _b123 = vget_high_s8(_b1);
                    int16x8_t _s = vmull_s8(_b001, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, _b023, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    _s = vmull_s8(_b101, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, _b123, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)));
                    _sum1 = vpadalq_s16(_sum1, _s);
                    _s = vmull_s8(_b001, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _s = vmlal_s8(_s, _b023, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)));
                    _sum2 = vpadalq_s16(_sum2, _s);
                    _s = vmull_s8(_b101, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _s = vmlal_s8(_s, _b123, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)));
                    _sum3 = vpadalq_s16(_sum3, _s);
                    pA += 8;
                    pB0 += 16;
                    pB1 += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x16_t _b = vcombine_s8(vld1_s8(pB0), vld1_s8(pB1));
                    int8x8_t _b0 = vget_low_s8(_b);
                    int8x8_t _b1 = vget_high_s8(_b);
                    int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    pA += 4;
                    pB0 += 8;
                    pB1 += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = vreinterpret_s8_s32(vld1_lane_s32((const int*)pB1, vld1_dup_s32((const int*)pB0), 1));
                    int8x8_t _a = vreinterpret_s8_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0));
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_high_s16(_p0)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p1)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_high_s16(_p1)));
                    pA += 2;
                    pB0 += 4;
                    pB1 += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales0);
                float32x4_t _bd1 = vld1q_f32(pB_descales1);
                float32x2_t _ad = vld1_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_bd0, _ad, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_bd1, _ad, 0));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_lane_f32(_bd0, _ad, 1));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_lane_f32(_bd1, _ad, 1));

                pA_descales += 2;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            vst1q_f32(outptr, _fsum2);
            outptr += 4;
            vst1q_f32(outptr, _fsum3);
            outptr += 4;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __aarch64__
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
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = vld1q_s8(pB + 0);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
                    int8x16_t _a0 = vld1q_s8(pA);
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
                    pA += 16;
                    pB += 32;
                }
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
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x16_t _b = vld1q_s8(pB);
                    _sum = vmmlaq_s32(_sum, _a, _b);
                    pA += 16;
                    pB += 16;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
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
                    int8x8_t _a0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _a1 = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pA + 4)));
                    int16x4_t _b = vreinterpret_s16_s8(vld1_s8(pB));
                    int8x8_t _b0 = vreinterpret_s8_s16(vdup_lane_s16(_b, 0));
                    int8x8_t _b1 = vreinterpret_s8_s16(vdup_lane_s16(_b, 1));
                    int8x8_t _b2 = vreinterpret_s8_s16(vdup_lane_s16(_b, 2));
                    int8x8_t _b3 = vreinterpret_s8_s16(vdup_lane_s16(_b, 3));
                    int32x4_t _s0 = vpaddlq_s16(vmull_s8(_a0, _b0));
                    int32x4_t _s1 = vpaddlq_s16(vmull_s8(_a0, _b1));
                    _s0 = vpadalq_s16(_s0, vmull_s8(_a1, _b2));
                    _s1 = vpadalq_s16(_s1, vmull_s8(_a1, _b3));
                    _sum = vaddq_s32(_sum, vzipq_s32(_s0, _s1).val[0]);
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
                fsum01 = outptr[1];
                fsum10 = outptr[2];
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
            outptr[0] = fsum01;
            outptr++;
            outptr[0] = fsum10;
            outptr++;
            outptr[0] = fsum11;
            outptr++;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
#endif // __ARM_NEON
        }
        for (; jj < max_jj; jj++)
        {
#if __ARM_NEON
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float32x4_t _fsum;

            if (k == 0)
                _fsum = vdupq_n_f32(0.f);
            else
                _fsum = vcombine_f32(vld1_f32(outptr), vdup_n_f32(0.f));

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = vld1_s8(pB);
                    int8x16_t _bb = vcombine_s8(_b, _b);
                    int32x4_t _s = vdotq_s32(vdupq_n_s32(0), _a, _bb);
                    _sum = vaddq_s32(_sum, vpaddq_s32(_s, _s));
                    pA += 16;
                    pB += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x16_t _aa = vcombine_s8(_a, vdup_n_s8(0));
                    int8x16_t _b = vreinterpretq_s8_s32(vld1q_dup_s32((const int*)pB));
                    _sum = vdotq_s32(_sum, _aa, _b);
                    pA += 8;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _a1 = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pA + 4)));
                    int16x4_t _b = vreinterpret_s16_s32(vld1_dup_s32((const int*)pB));
                    int8x8_t _b0 = vreinterpret_s8_s16(vdup_lane_s16(_b, 0));
                    int8x8_t _b1 = vreinterpret_s8_s16(vdup_lane_s16(_b, 1));
                    _sum = vaddq_s32(_sum, vpaddlq_s16(vmull_s8(_a0, _b0)));
                    _sum = vaddq_s32(_sum, vpaddlq_s16(vmull_s8(_a1, _b1)));
                    pA += 8;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    _sum = vaddq_s32(_sum, vpaddlq_s16(vmull_s8(_a, _b)));
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                    int16x8_t _s = vmull_s8(_a, vld1_dup_s8(pB));
                    _sum = vaddw_s16(_sum, vget_low_s16(_s));
                    pA += 2;
                    pB++;
                }

                float32x2_t _ad = vld1_f32(pA_descales);
                float32x4_t _scale = vmulq_n_f32(vcombine_f32(_ad, _ad), pB_descales[0]);
                _fsum = vmlaq_f32(_fsum, vcvtq_f32_s32(_sum), _scale);
                pA_descales += 2;
                pB_descales++;
            }

            vst1_f32(outptr, vget_low_f32(_fsum));
            outptr += 2;
            pB_panel += K;
            pB_descales_panel += block_count;
#else
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
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

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum00 = 0;
                int sum10 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const int b00 = pB[0];
                    const int b01 = pB[1];
                    sum00 += pA[0] * b00 + pA[2] * b01;
                    sum10 += pA[1] * b00 + pA[3] * b01;
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const int b0 = pB[0];
                    sum00 += pA[0] * b0;
                    sum10 += pA[1] * b0;
                    pA += 2;
                    pB += 1;
                }

                const float bd0 = pB_descales[0];
                const float ad0 = pA_descales[0];
                fsum00 += sum00 * ad0 * bd0;
                const float ad1 = pA_descales[1];
                fsum10 += sum10 * ad1 * bd0;

                pA_descales += 2;
                pB_descales += 1;
            }

            outptr[0] = fsum00;
            outptr++;
            outptr[0] = fsum10;
            outptr++;
            pB_panel += K;
            pB_descales_panel += block_count;
#endif // __ARM_NEON
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
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB_panel + (size_t)4 * k;
            const signed char* pB1 = pB_panel + (size_t)4 * K + (size_t)4 * k;
            const float* pB_descales0 = pB_descales_panel + (size_t)4 * block_start;
            const float* pB_descales1 = pB_descales_panel + (size_t)4 * block_count + (size_t)4 * block_start;
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
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = vld1q_s8(pB0);
                    int8x16_t _b1 = vld1q_s8(pB0 + 16);
                    int8x16_t _b2 = vld1q_s8(pB1);
                    int8x16_t _b3 = vld1q_s8(pB1 + 16);
                    int8x16_t _a0 = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
                    _msum2 = vmmlaq_s32(_msum2, _a0, _b2);
                    _msum3 = vmmlaq_s32(_msum3, _a0, _b3);
                    pA += 8;
                    pB0 += 32;
                    pB1 += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_low_s32(_msum2), vget_low_s32(_msum3));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = vld1q_s8(pB0);
                    int8x16_t _b1 = vld1q_s8(pB1);
                    int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    _sum1 = vdotq_s32(_sum1, _b1, _a0);
                    pA += 4;
                    pB0 += 16;
                    pB1 += 16;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    int8x16_t _b0 = vld1q_s8(pB0);
                    int8x16_t _b1 = vld1q_s8(pB1);
                    int16x8_t _s = vmull_s8(vget_low_s8(_b0), vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, vget_high_s8(_b0), vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    _s = vmull_s8(vget_low_s8(_b1), vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, vget_high_s8(_b1), vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _sum1 = vpadalq_s16(_sum1, _s);
                    pA += 4;
                    pB0 += 16;
                    pB1 += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x16_t _b = vcombine_s8(vld1_s8(pB0), vld1_s8(pB1));
                    int8x8_t _b0 = vget_low_s8(_b);
                    int8x8_t _b1 = vget_high_s8(_b);
                    int8x8_t _a0 = vreinterpret_s8_s16(vdup_lane_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, _a0)));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b1, _a0)));
                    pA += 2;
                    pB0 += 8;
                    pB1 += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = vreinterpret_s8_s32(vld1_lane_s32((const int*)pB1, vld1_dup_s32((const int*)pB0), 1));
                    int8x8_t _a0 = vld1_lane_s8(pA, vdup_n_s8(0), 0);
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_high_s16(_p0)));
                    pA++;
                    pB0 += 4;
                    pB1 += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales0);
                float32x4_t _bd1 = vld1q_f32(pB_descales1);
                const float _ad0 = pA_descales[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, _ad0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_bd1, _ad0));

                pA_descales++;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __aarch64__
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
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = vld1q_s8(pB + 0);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
                    int8x16_t _a0 = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
                    pA += 8;
                    pB += 32;
                }
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
#if __ARM_NEON
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            float32x4_t _fsum0;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vcombine_f32(vld1_f32(outptr), vdup_n_f32(0.f));
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
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = vld1q_s8(pB + 0);
                    int8x16_t _a0 = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    pA += 8;
                    pB += 16;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vdup_n_s32(0));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = vcombine_s8(vld1_s8(pB), vdup_n_s8(0));
                    int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    pA += 4;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    int8x8_t _b = vld1_s8(pB);
                    int32x2_t _b02 = vreinterpret_s32_s8(_b);
                    int8x8_t _b01 = vreinterpret_s8_s32(vdup_lane_s32(_b02, 0));
                    int8x8_t _b23 = vreinterpret_s8_s32(vdup_lane_s32(_b02, 1));
                    int16x8_t _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    pA += 4;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    int8x8_t _a0 = vreinterpret_s8_s16(vdup_lane_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, _a0)));
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB)));
                    int8x8_t _a0 = vld1_lane_s8(pA, vdup_n_s8(0), 0);
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    pA++;
                    pB += 2;
                }

                float32x4_t _bd0 = vcombine_f32(vld1_f32(pB_descales), vdup_n_f32(0.f));
                const float _ad0 = pA_descales[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, _ad0));

                pA_descales++;
                pB_descales += 2;
            }

            vst1_f32(outptr, vget_low_f32(_fsum0));
            outptr += 2;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
#else // __ARM_NEON
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
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

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum00 = 0;
                int sum01 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const int b00 = pB[0];
                    const int b01 = pB[1];
                    const int b10 = pB[2];
                    const int b11 = pB[3];
                    sum00 += pA[0] * b00 + pA[1] * b01;
                    sum01 += pA[0] * b10 + pA[1] * b11;
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const int b0 = pB[0];
                    const int b1 = pB[1];
                    sum00 += pA[0] * b0;
                    sum01 += pA[0] * b1;
                    pA++;
                    pB += 2;
                }

                const float bd0 = pB_descales[0];
                const float bd1 = pB_descales[1];
                const float ad0 = pA_descales[0];
                fsum00 += sum00 * ad0 * bd0;
                fsum01 += sum01 * ad0 * bd1;

                pA_descales++;
                pB_descales += 2;
            }

            outptr[0] = fsum00;
            outptr++;
            outptr[0] = fsum01;
            outptr++;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
#endif // __ARM_NEON
        }
        for (; jj < max_jj; jj++)
        {
#if __ARM_NEON
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            float32x4_t _fsum0;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vsetq_lane_f32(outptr[0], vdupq_n_f32(0.f), 0);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)pB)), vdup_n_s8(0));
                    int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    pA += 4;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    int8x8_t _b01 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                    int8x8_t _b23 = vext_s8(_b01, _b01, 2);
                    int16x8_t _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    pA += 4;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = vset_lane_s8(pB[0], vdup_n_s8(0), 0);
                    int8x8_t _a0 = vld1_lane_s8(pA, vdup_n_s8(0), 0);
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    pA++;
                    pB += 1;
                }

                float32x4_t _bd0 = vdupq_n_f32(pB_descales[0]);
                const float _ad0 = pA_descales[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, _ad0));

                pA_descales++;
                pB_descales += 1;
            }

            vst1q_lane_f32(outptr, _fsum0, 0);
            outptr++;
            pB_panel += K;
            pB_descales_panel += block_count;
#else  // __ARM_NEON
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
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    const int b0 = pB[0];
                    sum00 += pA[0] * b0;
                    pA++;
                    pB += 1;
                }

                const float bd0 = pB_descales[0];
                const float ad0 = pA_descales[0];
                fsum00 += sum00 * ad0 * bd0;

                pA_descales++;
                pB_descales += 1;
            }

            outptr[0] = fsum00;
            outptr++;
            pB_panel += K;
            pB_descales_panel += block_count;
#endif // __ARM_NEON
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
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

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
        if (constant_TILE_K < block_size)
            TILE_K = block_size;
        else
            TILE_K = constant_TILE_K / block_size * block_size;
        TILE_K = std::min(TILE_K, K);
    }
}
